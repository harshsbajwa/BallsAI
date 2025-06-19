import React, { useState } from "react";
import {
  View,
  Text,
  FlatList,
  TextInput,
  TouchableOpacity,
} from "react-native";
import { useQuery } from "@tanstack/react-query";
import { useRouter } from "expo-router";
import { apiClient } from "@/lib/api-client";
import { Loading } from "@/components/Loading";
import { Player } from "@nba/api/types";

export default function PlayersScreen() {
  const [searchQuery, setSearchQuery] = useState("");
  const router = useRouter();

  const { data: searchResults, isLoading, error } = useQuery({
    queryKey: ["players", "search", searchQuery],
    queryFn: () => apiClient.searchPlayers(searchQuery, 20),
    enabled: searchQuery.length >= 2,
  });

  const handlePlayerPress = (player: Player) => {
    router.push(`/player/${player.person_id}`);
  };

  return (
    <View className="flex-1 bg-gray-50">
      <View className="bg-white p-4 shadow-sm">
        <TextInput
          className="border border-gray-300 rounded-lg px-4 py-3 text-base"
          placeholder="Search for players..."
          value={searchQuery}
          onChangeText={setSearchQuery}
          autoCapitalize="words"
        />
      </View>

      <View className="flex-1 p-4">
        {searchQuery.length < 2 ? (
          <View className="flex-1 items-center justify-center">
            <Text className="text-gray-500 text-lg">Search for NBA players</Text>
            <Text className="text-gray-400 text-sm mt-2">
              Enter at least 2 characters
            </Text>
          </View>
        ) : isLoading ? (
          <Loading text="Searching players..." />
        ) : error ? (
          <View className="flex-1 items-center justify-center">
            <Text className="text-red-500 text-lg">Error loading players</Text>
            <Text className="text-gray-500 text-sm mt-2">Please try again</Text>
          </View>
        ) : searchResults?.items.length === 0 ? (
          <View className="flex-1 items-center justify-center">
            <Text className="text-gray-500 text-lg">No players found</Text>
            <Text className="text-gray-400 text-sm mt-2">
              Try a different search term
            </Text>
          </View>
        ) : (
          <FlatList
            data={searchResults?.items}
            keyExtractor={(item) => item.person_id.toString()}
            renderItem={({ item }) => (
              <TouchableOpacity
                onPress={() => handlePlayerPress(item)}
                className="bg-white rounded-lg p-4 mb-3 shadow-sm"
              >
                <Text className="font-bold text-lg text-gray-900">
                  {item.full_name}
                </Text>
                <View className="flex-row justify-between items-center mt-2">
                  <Text className="text-gray-600">
                    {item.primary_position}
                  </Text>
                  {item.draft_year && (
                    <Text className="text-gray-500 text-sm">
                      Draft {item.draft_year}
                    </Text>
                  )}
                </View>
              </TouchableOpacity>
            )}
            showsVerticalScrollIndicator={false}
          />
        )}
      </View>
    </View>
  );
}